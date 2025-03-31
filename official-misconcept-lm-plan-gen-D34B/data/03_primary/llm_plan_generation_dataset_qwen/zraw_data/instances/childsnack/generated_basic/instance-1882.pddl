; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 307167

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child318 child405 - child
    bread105 bread264 - bread-portion
    content6 content472 - content-portion
    tray441 tray19 - tray
    table440 table486 table442 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray441 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread105)
     (at_kitchen_bread bread264)
     (at_kitchen_content content6)
     (at_kitchen_content content472)
     (no_gluten_bread bread105)
     (no_gluten_content content472)
     (allergic_gluten child405)
     (not_allergic_gluten child318)
     (waiting child318 table442)
     (waiting child405 table440)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child318)
     (served child405)
    )
  )
)
