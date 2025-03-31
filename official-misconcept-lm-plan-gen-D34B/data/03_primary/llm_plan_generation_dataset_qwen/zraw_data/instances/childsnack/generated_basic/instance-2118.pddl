; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 532598

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child56 child23 - child
    bread417 bread272 - bread-portion
    content227 content381 - content-portion
    tray122 tray221 tray135 - tray
    table255 table399 table85 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray122 kitchen)
     (at tray221 kitchen)
     (at tray135 kitchen)
     (at_kitchen_bread bread417)
     (at_kitchen_bread bread272)
     (at_kitchen_content content227)
     (at_kitchen_content content381)
     (no_gluten_bread bread417)
     (no_gluten_content content381)
     (allergic_gluten child23)
     (not_allergic_gluten child56)
     (waiting child56 table85)
     (waiting child23 table255)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child56)
     (served child23)
    )
  )
)
