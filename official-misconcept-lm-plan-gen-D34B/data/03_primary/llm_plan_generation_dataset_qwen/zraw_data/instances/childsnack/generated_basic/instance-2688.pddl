; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 74575

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child174 child333 child469 - child
    bread266 bread475 bread117 - bread-portion
    content128 content133 content478 - content-portion
    tray466 tray358 tray331 - tray
    table450 table425 table378 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray466 kitchen)
     (at tray358 kitchen)
     (at tray331 kitchen)
     (at_kitchen_bread bread266)
     (at_kitchen_bread bread475)
     (at_kitchen_bread bread117)
     (at_kitchen_content content128)
     (at_kitchen_content content133)
     (at_kitchen_content content478)
     (not_allergic_gluten child469)
     (not_allergic_gluten child174)
     (not_allergic_gluten child333)
     (waiting child174 table425)
     (waiting child333 table425)
     (waiting child469 table450)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child174)
     (served child333)
     (served child469)
    )
  )
)
