; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 977491

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child190 child283 - child
    bread45 bread159 - bread-portion
    content274 content491 - content-portion
    tray303 tray269 - tray
    table101 table476 table116 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray303 kitchen)
     (at tray269 kitchen)
     (at_kitchen_bread bread45)
     (at_kitchen_bread bread159)
     (at_kitchen_content content274)
     (at_kitchen_content content491)
     (not_allergic_gluten child190)
     (not_allergic_gluten child283)
     (waiting child190 table101)
     (waiting child283 table101)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child190)
     (served child283)
    )
  )
)
