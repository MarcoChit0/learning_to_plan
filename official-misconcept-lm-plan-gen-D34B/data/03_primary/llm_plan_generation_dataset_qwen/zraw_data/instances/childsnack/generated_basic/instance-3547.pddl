; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 842540

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child401 child351 - child
    bread159 bread80 - bread-portion
    content332 content313 - content-portion
    tray275 tray194 - tray
    table488 table263 table445 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray275 kitchen)
     (at tray194 kitchen)
     (at_kitchen_bread bread159)
     (at_kitchen_bread bread80)
     (at_kitchen_content content332)
     (at_kitchen_content content313)
     (not_allergic_gluten child401)
     (not_allergic_gluten child351)
     (waiting child401 table263)
     (waiting child351 table488)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child401)
     (served child351)
    )
  )
)
