; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 882059

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 child264 - child
    bread122 bread261 - bread-portion
    content139 content186 - content-portion
    tray231 - tray
    table428 table215 table410 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray231 kitchen)
     (at_kitchen_bread bread122)
     (at_kitchen_bread bread261)
     (at_kitchen_content content139)
     (at_kitchen_content content186)
     (not_allergic_gluten child264)
     (not_allergic_gluten child73)
     (waiting child73 table428)
     (waiting child264 table410)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child73)
     (served child264)
    )
  )
)
