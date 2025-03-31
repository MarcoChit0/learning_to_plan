; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 458933

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child217 child264 - child
    bread388 bread120 - bread-portion
    content331 content298 - content-portion
    tray77 tray336 tray102 - tray
    table242 table179 table471 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray77 kitchen)
     (at tray336 kitchen)
     (at tray102 kitchen)
     (at_kitchen_bread bread388)
     (at_kitchen_bread bread120)
     (at_kitchen_content content331)
     (at_kitchen_content content298)
     (not_allergic_gluten child217)
     (not_allergic_gluten child264)
     (waiting child217 table242)
     (waiting child264 table471)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child217)
     (served child264)
    )
  )
)
