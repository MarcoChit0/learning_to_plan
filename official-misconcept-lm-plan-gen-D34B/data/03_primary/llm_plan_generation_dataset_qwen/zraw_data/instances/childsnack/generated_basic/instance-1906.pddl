; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 710093

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child23 - child
    bread89 - bread-portion
    content445 - content-portion
    tray203 tray161 tray103 - tray
    table117 table154 table108 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray203 kitchen)
     (at tray161 kitchen)
     (at tray103 kitchen)
     (at_kitchen_bread bread89)
     (at_kitchen_content content445)
     (not_allergic_gluten child23)
     (waiting child23 table154)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child23)
    )
  )
)
