; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 623880

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child82 child179 - child
    bread60 bread386 - bread-portion
    content465 content163 - content-portion
    tray16 - tray
    table199 table228 table315 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray16 kitchen)
     (at_kitchen_bread bread60)
     (at_kitchen_bread bread386)
     (at_kitchen_content content465)
     (at_kitchen_content content163)
     (no_gluten_bread bread60)
     (no_gluten_content content465)
     (allergic_gluten child82)
     (not_allergic_gluten child179)
     (waiting child82 table199)
     (waiting child179 table315)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child82)
     (served child179)
    )
  )
)
