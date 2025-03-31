; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 479077

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child38 child483 - child
    bread223 bread449 - bread-portion
    content136 content104 - content-portion
    tray381 - tray
    table7 table264 table168 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray381 kitchen)
     (at_kitchen_bread bread223)
     (at_kitchen_bread bread449)
     (at_kitchen_content content136)
     (at_kitchen_content content104)
     (no_gluten_bread bread449)
     (no_gluten_content content104)
     (allergic_gluten child483)
     (not_allergic_gluten child38)
     (waiting child38 table7)
     (waiting child483 table7)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child38)
     (served child483)
    )
  )
)
