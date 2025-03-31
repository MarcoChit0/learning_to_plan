; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 131819

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child29 child287 - child
    bread290 bread344 - bread-portion
    content75 content238 - content-portion
    tray404 tray160 - tray
    table481 table28 table170 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray404 kitchen)
     (at tray160 kitchen)
     (at_kitchen_bread bread290)
     (at_kitchen_bread bread344)
     (at_kitchen_content content75)
     (at_kitchen_content content238)
     (no_gluten_bread bread344)
     (no_gluten_content content75)
     (allergic_gluten child287)
     (not_allergic_gluten child29)
     (waiting child29 table28)
     (waiting child287 table28)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child29)
     (served child287)
    )
  )
)
