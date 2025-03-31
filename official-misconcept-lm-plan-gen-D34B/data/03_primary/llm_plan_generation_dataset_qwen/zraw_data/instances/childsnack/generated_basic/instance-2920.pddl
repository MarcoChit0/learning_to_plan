; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 706511

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child224 child42 - child
    bread55 bread161 - bread-portion
    content221 content344 - content-portion
    tray56 tray154 tray216 - tray
    table370 table445 table71 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray56 kitchen)
     (at tray154 kitchen)
     (at tray216 kitchen)
     (at_kitchen_bread bread55)
     (at_kitchen_bread bread161)
     (at_kitchen_content content221)
     (at_kitchen_content content344)
     (no_gluten_bread bread161)
     (no_gluten_content content221)
     (allergic_gluten child224)
     (not_allergic_gluten child42)
     (waiting child224 table445)
     (waiting child42 table370)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child224)
     (served child42)
    )
  )
)
