; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 931327

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child436 - child
    bread137 - bread-portion
    content224 - content-portion
    tray293 - tray
    table129 table15 table120 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray293 kitchen)
     (at_kitchen_bread bread137)
     (at_kitchen_content content224)
     (not_allergic_gluten child436)
     (waiting child436 table129)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child436)
    )
  )
)
