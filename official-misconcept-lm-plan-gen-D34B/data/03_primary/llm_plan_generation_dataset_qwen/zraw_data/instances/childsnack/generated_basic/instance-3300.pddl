; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 728704

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 child1 - child
    bread411 bread247 - bread-portion
    content324 content471 - content-portion
    tray495 tray403 tray9 tray11 - tray
    table73 table302 table37 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray495 kitchen)
     (at tray403 kitchen)
     (at tray9 kitchen)
     (at tray11 kitchen)
     (at_kitchen_bread bread411)
     (at_kitchen_bread bread247)
     (at_kitchen_content content324)
     (at_kitchen_content content471)
     (no_gluten_bread bread247)
     (no_gluten_content content471)
     (allergic_gluten child1)
     (not_allergic_gluten child323)
     (waiting child323 table302)
     (waiting child1 table37)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child323)
     (served child1)
    )
  )
)
