; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 489092

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child101 child266 - child
    bread225 bread333 - bread-portion
    content43 content327 - content-portion
    tray71 tray192 tray132 - tray
    table162 table109 table160 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray71 kitchen)
     (at tray192 kitchen)
     (at tray132 kitchen)
     (at_kitchen_bread bread225)
     (at_kitchen_bread bread333)
     (at_kitchen_content content43)
     (at_kitchen_content content327)
     (not_allergic_gluten child101)
     (not_allergic_gluten child266)
     (waiting child101 table160)
     (waiting child266 table162)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child101)
     (served child266)
    )
  )
)
