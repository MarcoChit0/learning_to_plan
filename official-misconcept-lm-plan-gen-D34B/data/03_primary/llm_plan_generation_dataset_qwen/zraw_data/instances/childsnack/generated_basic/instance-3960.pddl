; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 96810

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child393 child81 - child
    bread52 bread350 - bread-portion
    content20 content191 - content-portion
    tray29 - tray
    table325 table445 table159 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray29 kitchen)
     (at_kitchen_bread bread52)
     (at_kitchen_bread bread350)
     (at_kitchen_content content20)
     (at_kitchen_content content191)
     (no_gluten_bread bread350)
     (no_gluten_content content191)
     (allergic_gluten child81)
     (not_allergic_gluten child393)
     (waiting child393 table445)
     (waiting child81 table325)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child393)
     (served child81)
    )
  )
)
