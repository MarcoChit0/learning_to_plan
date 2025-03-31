; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 694565

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child220 - child
    bread41 - bread-portion
    content150 - content-portion
    tray139 - tray
    table93 table267 table445 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray139 kitchen)
     (at_kitchen_bread bread41)
     (at_kitchen_content content150)
     (not_allergic_gluten child220)
     (waiting child220 table445)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child220)
    )
  )
)
