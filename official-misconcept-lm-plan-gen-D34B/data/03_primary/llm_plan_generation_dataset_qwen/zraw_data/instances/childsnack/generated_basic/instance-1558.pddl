; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 595478

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child240 child4 - child
    bread78 bread473 - bread-portion
    content52 content40 - content-portion
    tray308 tray314 tray167 tray312 - tray
    table262 table391 table122 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray308 kitchen)
     (at tray314 kitchen)
     (at tray167 kitchen)
     (at tray312 kitchen)
     (at_kitchen_bread bread78)
     (at_kitchen_bread bread473)
     (at_kitchen_content content52)
     (at_kitchen_content content40)
     (not_allergic_gluten child240)
     (not_allergic_gluten child4)
     (waiting child240 table122)
     (waiting child4 table391)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child240)
     (served child4)
    )
  )
)
