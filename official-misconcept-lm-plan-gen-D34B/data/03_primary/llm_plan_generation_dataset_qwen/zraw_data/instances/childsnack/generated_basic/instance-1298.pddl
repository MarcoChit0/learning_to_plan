; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 648142

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child35 child271 - child
    bread340 bread238 - bread-portion
    content482 content476 - content-portion
    tray214 tray117 tray480 tray45 - tray
    table50 table255 table167 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray214 kitchen)
     (at tray117 kitchen)
     (at tray480 kitchen)
     (at tray45 kitchen)
     (at_kitchen_bread bread340)
     (at_kitchen_bread bread238)
     (at_kitchen_content content482)
     (at_kitchen_content content476)
     (no_gluten_bread bread238)
     (no_gluten_content content476)
     (allergic_gluten child271)
     (not_allergic_gluten child35)
     (waiting child35 table255)
     (waiting child271 table255)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child35)
     (served child271)
    )
  )
)
