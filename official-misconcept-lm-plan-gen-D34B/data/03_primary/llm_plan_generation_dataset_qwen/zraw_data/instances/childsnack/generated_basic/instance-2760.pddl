; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 297638

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child46 child472 - child
    bread357 bread287 - bread-portion
    content211 content465 - content-portion
    tray447 - tray
    table424 table366 table91 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray447 kitchen)
     (at_kitchen_bread bread357)
     (at_kitchen_bread bread287)
     (at_kitchen_content content211)
     (at_kitchen_content content465)
     (no_gluten_bread bread357)
     (no_gluten_content content465)
     (allergic_gluten child46)
     (not_allergic_gluten child472)
     (waiting child46 table366)
     (waiting child472 table366)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child46)
     (served child472)
    )
  )
)
