; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 264436

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child163 child368 child221 - child
    bread264 bread110 bread131 - bread-portion
    content434 content378 content59 - content-portion
    tray65 - tray
    table71 table181 table156 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray65 kitchen)
     (at_kitchen_bread bread264)
     (at_kitchen_bread bread110)
     (at_kitchen_bread bread131)
     (at_kitchen_content content434)
     (at_kitchen_content content378)
     (at_kitchen_content content59)
     (not_allergic_gluten child368)
     (not_allergic_gluten child163)
     (not_allergic_gluten child221)
     (waiting child163 table71)
     (waiting child368 table156)
     (waiting child221 table71)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child163)
     (served child368)
     (served child221)
    )
  )
)
