; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 665060

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child447 child246 - child
    bread139 bread290 - bread-portion
    content492 content222 - content-portion
    tray424 tray29 tray122 - tray
    table99 table432 table98 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray424 kitchen)
     (at tray29 kitchen)
     (at tray122 kitchen)
     (at_kitchen_bread bread139)
     (at_kitchen_bread bread290)
     (at_kitchen_content content492)
     (at_kitchen_content content222)
     (no_gluten_bread bread290)
     (no_gluten_content content222)
     (allergic_gluten child447)
     (not_allergic_gluten child246)
     (waiting child447 table99)
     (waiting child246 table98)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child447)
     (served child246)
    )
  )
)
