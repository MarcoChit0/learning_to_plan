; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 790271

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child206 child462 - child
    bread223 bread225 - bread-portion
    content340 content318 - content-portion
    tray255 tray312 - tray
    table473 table283 table242 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray255 kitchen)
     (at tray312 kitchen)
     (at_kitchen_bread bread223)
     (at_kitchen_bread bread225)
     (at_kitchen_content content340)
     (at_kitchen_content content318)
     (no_gluten_bread bread223)
     (no_gluten_content content340)
     (allergic_gluten child462)
     (not_allergic_gluten child206)
     (waiting child206 table473)
     (waiting child462 table473)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child206)
     (served child462)
    )
  )
)
