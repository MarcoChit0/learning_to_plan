; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 739078

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child229 child216 - child
    bread476 bread276 - bread-portion
    content329 content36 - content-portion
    tray460 tray53 tray217 - tray
    table334 table434 table477 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray460 kitchen)
     (at tray53 kitchen)
     (at tray217 kitchen)
     (at_kitchen_bread bread476)
     (at_kitchen_bread bread276)
     (at_kitchen_content content329)
     (at_kitchen_content content36)
     (no_gluten_bread bread276)
     (no_gluten_content content329)
     (allergic_gluten child216)
     (not_allergic_gluten child229)
     (waiting child229 table434)
     (waiting child216 table334)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child229)
     (served child216)
    )
  )
)
