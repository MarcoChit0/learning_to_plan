; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 412775

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child450 child458 child264 - child
    bread54 bread250 bread45 - bread-portion
    content416 content310 content72 - content-portion
    tray238 - tray
    table231 table44 table304 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray238 kitchen)
     (at_kitchen_bread bread54)
     (at_kitchen_bread bread250)
     (at_kitchen_bread bread45)
     (at_kitchen_content content416)
     (at_kitchen_content content310)
     (at_kitchen_content content72)
     (not_allergic_gluten child458)
     (not_allergic_gluten child264)
     (not_allergic_gluten child450)
     (waiting child450 table44)
     (waiting child458 table44)
     (waiting child264 table44)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child450)
     (served child458)
     (served child264)
    )
  )
)
