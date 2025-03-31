; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 479847

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child461 child211 - child
    bread405 bread154 - bread-portion
    content378 content303 - content-portion
    tray476 tray211 tray480 - tray
    table208 table320 table318 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray476 kitchen)
     (at tray211 kitchen)
     (at tray480 kitchen)
     (at_kitchen_bread bread405)
     (at_kitchen_bread bread154)
     (at_kitchen_content content378)
     (at_kitchen_content content303)
     (no_gluten_bread bread405)
     (no_gluten_content content303)
     (allergic_gluten child211)
     (not_allergic_gluten child461)
     (waiting child461 table320)
     (waiting child211 table318)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child461)
     (served child211)
    )
  )
)
