; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 543062

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child247 child92 - child
    bread201 bread238 - bread-portion
    content53 content60 - content-portion
    tray120 tray43 tray264 tray325 - tray
    table246 table101 table497 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray120 kitchen)
     (at tray43 kitchen)
     (at tray264 kitchen)
     (at tray325 kitchen)
     (at_kitchen_bread bread201)
     (at_kitchen_bread bread238)
     (at_kitchen_content content53)
     (at_kitchen_content content60)
     (no_gluten_bread bread238)
     (no_gluten_content content53)
     (allergic_gluten child247)
     (not_allergic_gluten child92)
     (waiting child247 table246)
     (waiting child92 table246)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child247)
     (served child92)
    )
  )
)
