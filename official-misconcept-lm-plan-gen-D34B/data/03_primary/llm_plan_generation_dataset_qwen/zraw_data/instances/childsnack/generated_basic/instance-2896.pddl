; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 992832

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child122 child391 - child
    bread251 bread146 - bread-portion
    content203 content17 - content-portion
    tray8 tray354 tray494 - tray
    table453 table157 table190 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray8 kitchen)
     (at tray354 kitchen)
     (at tray494 kitchen)
     (at_kitchen_bread bread251)
     (at_kitchen_bread bread146)
     (at_kitchen_content content203)
     (at_kitchen_content content17)
     (no_gluten_bread bread146)
     (no_gluten_content content203)
     (allergic_gluten child122)
     (not_allergic_gluten child391)
     (waiting child122 table190)
     (waiting child391 table157)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child122)
     (served child391)
    )
  )
)
