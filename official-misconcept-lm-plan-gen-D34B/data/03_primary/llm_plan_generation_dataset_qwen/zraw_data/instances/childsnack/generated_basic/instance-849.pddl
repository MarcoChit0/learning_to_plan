; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 314551

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child51 child233 - child
    bread55 bread274 - bread-portion
    content486 content89 - content-portion
    tray106 tray6 - tray
    table65 table342 table256 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray106 kitchen)
     (at tray6 kitchen)
     (at_kitchen_bread bread55)
     (at_kitchen_bread bread274)
     (at_kitchen_content content486)
     (at_kitchen_content content89)
     (no_gluten_bread bread274)
     (no_gluten_content content486)
     (allergic_gluten child233)
     (not_allergic_gluten child51)
     (waiting child51 table256)
     (waiting child233 table65)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child51)
     (served child233)
    )
  )
)
