; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 39841

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child393 child106 - child
    bread37 bread4 - bread-portion
    content203 content347 - content-portion
    tray105 - tray
    table235 table222 table160 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray105 kitchen)
     (at_kitchen_bread bread37)
     (at_kitchen_bread bread4)
     (at_kitchen_content content203)
     (at_kitchen_content content347)
     (no_gluten_bread bread37)
     (no_gluten_content content347)
     (allergic_gluten child106)
     (not_allergic_gluten child393)
     (waiting child393 table222)
     (waiting child106 table222)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child393)
     (served child106)
    )
  )
)
