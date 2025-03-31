; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 688348

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child90 child85 - child
    bread162 bread328 - bread-portion
    content120 content247 - content-portion
    tray478 - tray
    table355 table219 table82 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray478 kitchen)
     (at_kitchen_bread bread162)
     (at_kitchen_bread bread328)
     (at_kitchen_content content120)
     (at_kitchen_content content247)
     (not_allergic_gluten child90)
     (not_allergic_gluten child85)
     (waiting child90 table219)
     (waiting child85 table355)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child90)
     (served child85)
    )
  )
)
