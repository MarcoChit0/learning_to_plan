; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 634632

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child229 child440 - child
    bread365 bread265 - bread-portion
    content78 content234 - content-portion
    tray128 tray379 tray197 tray196 - tray
    table480 table137 table223 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at tray379 kitchen)
     (at tray197 kitchen)
     (at tray196 kitchen)
     (at_kitchen_bread bread365)
     (at_kitchen_bread bread265)
     (at_kitchen_content content78)
     (at_kitchen_content content234)
     (no_gluten_bread bread365)
     (no_gluten_content content234)
     (allergic_gluten child229)
     (not_allergic_gluten child440)
     (waiting child229 table480)
     (waiting child440 table480)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child229)
     (served child440)
    )
  )
)
