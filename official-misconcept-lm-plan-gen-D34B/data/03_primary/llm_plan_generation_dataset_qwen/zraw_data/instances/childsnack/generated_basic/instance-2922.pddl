; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 331321

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child477 child175 - child
    bread148 bread394 - bread-portion
    content295 content219 - content-portion
    tray241 tray48 tray375 - tray
    table289 table313 table373 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at tray48 kitchen)
     (at tray375 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_bread bread394)
     (at_kitchen_content content295)
     (at_kitchen_content content219)
     (no_gluten_bread bread148)
     (no_gluten_content content295)
     (allergic_gluten child175)
     (not_allergic_gluten child477)
     (waiting child477 table373)
     (waiting child175 table313)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child477)
     (served child175)
    )
  )
)
