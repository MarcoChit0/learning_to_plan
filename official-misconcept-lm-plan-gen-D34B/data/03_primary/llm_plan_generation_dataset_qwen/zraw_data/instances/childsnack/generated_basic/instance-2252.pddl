; child-snack task with 3 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 989929

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child210 child278 child244 - child
    bread351 bread180 bread45 - bread-portion
    content175 content496 content125 - content-portion
    tray38 tray491 tray209 tray145 - tray
    table361 table313 table315 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray38 kitchen)
     (at tray491 kitchen)
     (at tray209 kitchen)
     (at tray145 kitchen)
     (at_kitchen_bread bread351)
     (at_kitchen_bread bread180)
     (at_kitchen_bread bread45)
     (at_kitchen_content content175)
     (at_kitchen_content content496)
     (at_kitchen_content content125)
     (no_gluten_bread bread180)
     (no_gluten_content content175)
     (allergic_gluten child278)
     (not_allergic_gluten child244)
     (not_allergic_gluten child210)
     (waiting child210 table313)
     (waiting child278 table313)
     (waiting child244 table361)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child210)
     (served child278)
     (served child244)
    )
  )
)
