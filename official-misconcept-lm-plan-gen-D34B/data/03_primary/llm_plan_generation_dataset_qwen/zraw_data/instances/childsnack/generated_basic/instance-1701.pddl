; child-snack task with 3 children and 0.57 gluten factor 
; constant factor of 1.3
; random seed: 861977

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child369 child279 child352 - child
    bread15 bread459 bread160 - bread-portion
    content190 content18 content267 - content-portion
    tray248 tray38 tray46 tray29 - tray
    table144 table320 table468 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray248 kitchen)
     (at tray38 kitchen)
     (at tray46 kitchen)
     (at tray29 kitchen)
     (at_kitchen_bread bread15)
     (at_kitchen_bread bread459)
     (at_kitchen_bread bread160)
     (at_kitchen_content content190)
     (at_kitchen_content content18)
     (at_kitchen_content content267)
     (no_gluten_bread bread459)
     (no_gluten_content content267)
     (allergic_gluten child279)
     (not_allergic_gluten child369)
     (not_allergic_gluten child352)
     (waiting child369 table320)
     (waiting child279 table468)
     (waiting child352 table320)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child369)
     (served child279)
     (served child352)
    )
  )
)
