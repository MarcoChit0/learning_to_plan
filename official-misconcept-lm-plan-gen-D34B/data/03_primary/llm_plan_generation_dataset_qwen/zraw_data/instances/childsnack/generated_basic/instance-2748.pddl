; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 58785

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child212 child205 - child
    bread289 bread327 - bread-portion
    content433 content120 - content-portion
    tray246 - tray
    table222 table5 table27 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray246 kitchen)
     (at_kitchen_bread bread289)
     (at_kitchen_bread bread327)
     (at_kitchen_content content433)
     (at_kitchen_content content120)
     (no_gluten_bread bread327)
     (no_gluten_content content433)
     (allergic_gluten child212)
     (not_allergic_gluten child205)
     (waiting child212 table27)
     (waiting child205 table27)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child212)
     (served child205)
    )
  )
)
