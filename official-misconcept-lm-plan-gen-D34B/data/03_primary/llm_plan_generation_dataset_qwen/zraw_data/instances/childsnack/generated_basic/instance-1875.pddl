; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 836037

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child264 child355 - child
    bread258 bread204 - bread-portion
    content494 content184 - content-portion
    tray421 tray317 - tray
    table269 table397 table392 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray421 kitchen)
     (at tray317 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_bread bread204)
     (at_kitchen_content content494)
     (at_kitchen_content content184)
     (no_gluten_bread bread204)
     (no_gluten_content content494)
     (allergic_gluten child355)
     (not_allergic_gluten child264)
     (waiting child264 table392)
     (waiting child355 table397)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child264)
     (served child355)
    )
  )
)
