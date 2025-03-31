; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 938454

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child222 - child
    bread159 - bread-portion
    content167 - content-portion
    tray276 - tray
    table175 table155 table416 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray276 kitchen)
     (at_kitchen_bread bread159)
     (at_kitchen_content content167)
     (not_allergic_gluten child222)
     (waiting child222 table416)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child222)
    )
  )
)
