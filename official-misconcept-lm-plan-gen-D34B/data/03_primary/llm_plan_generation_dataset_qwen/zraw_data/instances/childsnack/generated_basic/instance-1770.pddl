; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 165473

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child83 - child
    bread176 - bread-portion
    content93 - content-portion
    tray341 - tray
    table156 table201 table416 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray341 kitchen)
     (at_kitchen_bread bread176)
     (at_kitchen_content content93)
     (not_allergic_gluten child83)
     (waiting child83 table201)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child83)
    )
  )
)
