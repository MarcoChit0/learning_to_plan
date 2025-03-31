; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 336603

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child88 child397 - child
    bread364 bread388 - bread-portion
    content300 content113 - content-portion
    tray371 tray445 tray8 - tray
    table301 table363 table214 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray371 kitchen)
     (at tray445 kitchen)
     (at tray8 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_bread bread388)
     (at_kitchen_content content300)
     (at_kitchen_content content113)
     (not_allergic_gluten child397)
     (not_allergic_gluten child88)
     (waiting child88 table301)
     (waiting child397 table301)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child88)
     (served child397)
    )
  )
)
