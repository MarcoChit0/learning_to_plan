; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 504750

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child16 child236 - child
    bread436 bread119 - bread-portion
    content260 content449 - content-portion
    tray47 - tray
    table94 table70 table85 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray47 kitchen)
     (at_kitchen_bread bread436)
     (at_kitchen_bread bread119)
     (at_kitchen_content content260)
     (at_kitchen_content content449)
     (no_gluten_bread bread119)
     (no_gluten_content content449)
     (allergic_gluten child16)
     (not_allergic_gluten child236)
     (waiting child16 table94)
     (waiting child236 table94)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child16)
     (served child236)
    )
  )
)
