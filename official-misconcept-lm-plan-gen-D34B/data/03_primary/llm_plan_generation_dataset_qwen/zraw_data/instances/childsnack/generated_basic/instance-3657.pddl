; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 753169

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child59 child349 - child
    bread313 bread467 - bread-portion
    content447 content496 - content-portion
    tray149 tray402 tray114 tray450 - tray
    table151 table161 table399 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray149 kitchen)
     (at tray402 kitchen)
     (at tray114 kitchen)
     (at tray450 kitchen)
     (at_kitchen_bread bread313)
     (at_kitchen_bread bread467)
     (at_kitchen_content content447)
     (at_kitchen_content content496)
     (no_gluten_bread bread313)
     (no_gluten_content content447)
     (allergic_gluten child59)
     (not_allergic_gluten child349)
     (waiting child59 table399)
     (waiting child349 table399)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child59)
     (served child349)
    )
  )
)
