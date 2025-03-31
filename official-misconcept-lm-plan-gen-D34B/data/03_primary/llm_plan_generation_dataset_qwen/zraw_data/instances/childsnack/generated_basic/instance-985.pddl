; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 281929

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child79 child321 child346 - child
    bread50 bread72 bread87 - bread-portion
    content198 content148 content290 - content-portion
    tray239 - tray
    table368 table397 table34 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray239 kitchen)
     (at_kitchen_bread bread50)
     (at_kitchen_bread bread72)
     (at_kitchen_bread bread87)
     (at_kitchen_content content198)
     (at_kitchen_content content148)
     (at_kitchen_content content290)
     (not_allergic_gluten child321)
     (not_allergic_gluten child346)
     (not_allergic_gluten child79)
     (waiting child79 table397)
     (waiting child321 table397)
     (waiting child346 table397)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child79)
     (served child321)
     (served child346)
    )
  )
)
