; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 254481

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child180 child55 - child
    bread43 bread276 - bread-portion
    content49 content207 - content-portion
    tray140 tray162 tray387 - tray
    table34 table277 table75 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray140 kitchen)
     (at tray162 kitchen)
     (at tray387 kitchen)
     (at_kitchen_bread bread43)
     (at_kitchen_bread bread276)
     (at_kitchen_content content49)
     (at_kitchen_content content207)
     (not_allergic_gluten child180)
     (not_allergic_gluten child55)
     (waiting child180 table75)
     (waiting child55 table277)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child180)
     (served child55)
    )
  )
)
