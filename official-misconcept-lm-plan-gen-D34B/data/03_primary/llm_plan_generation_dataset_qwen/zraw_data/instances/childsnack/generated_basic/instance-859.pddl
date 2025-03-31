; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 270403

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child152 child307 - child
    bread479 bread456 - bread-portion
    content174 content291 - content-portion
    tray436 tray139 - tray
    table25 table109 table351 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray436 kitchen)
     (at tray139 kitchen)
     (at_kitchen_bread bread479)
     (at_kitchen_bread bread456)
     (at_kitchen_content content174)
     (at_kitchen_content content291)
     (no_gluten_bread bread456)
     (no_gluten_content content174)
     (allergic_gluten child152)
     (not_allergic_gluten child307)
     (waiting child152 table351)
     (waiting child307 table351)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child152)
     (served child307)
    )
  )
)
