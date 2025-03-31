; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 156077

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child30 child389 - child
    bread208 bread11 - bread-portion
    content14 content429 - content-portion
    tray106 tray233 - tray
    table103 table212 table471 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray106 kitchen)
     (at tray233 kitchen)
     (at_kitchen_bread bread208)
     (at_kitchen_bread bread11)
     (at_kitchen_content content14)
     (at_kitchen_content content429)
     (no_gluten_bread bread208)
     (no_gluten_content content14)
     (allergic_gluten child30)
     (not_allergic_gluten child389)
     (waiting child30 table212)
     (waiting child389 table471)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child30)
     (served child389)
    )
  )
)
