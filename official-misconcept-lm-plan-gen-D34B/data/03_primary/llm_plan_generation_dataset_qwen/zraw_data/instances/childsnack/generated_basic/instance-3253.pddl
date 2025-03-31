; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 626156

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child7 child216 - child
    bread368 bread291 - bread-portion
    content283 content56 - content-portion
    tray97 tray201 tray63 tray107 - tray
    table235 table349 table215 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray97 kitchen)
     (at tray201 kitchen)
     (at tray63 kitchen)
     (at tray107 kitchen)
     (at_kitchen_bread bread368)
     (at_kitchen_bread bread291)
     (at_kitchen_content content283)
     (at_kitchen_content content56)
     (no_gluten_bread bread291)
     (no_gluten_content content56)
     (allergic_gluten child216)
     (not_allergic_gluten child7)
     (waiting child7 table215)
     (waiting child216 table349)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child7)
     (served child216)
    )
  )
)
