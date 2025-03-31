; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 320910

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child234 child404 - child
    bread174 bread155 - bread-portion
    content84 content411 - content-portion
    tray38 tray324 - tray
    table167 table146 table12 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray38 kitchen)
     (at tray324 kitchen)
     (at_kitchen_bread bread174)
     (at_kitchen_bread bread155)
     (at_kitchen_content content84)
     (at_kitchen_content content411)
     (no_gluten_bread bread155)
     (no_gluten_content content411)
     (allergic_gluten child234)
     (not_allergic_gluten child404)
     (waiting child234 table12)
     (waiting child404 table146)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child234)
     (served child404)
    )
  )
)
