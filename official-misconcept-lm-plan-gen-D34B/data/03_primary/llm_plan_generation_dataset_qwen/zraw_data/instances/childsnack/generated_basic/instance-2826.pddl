; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 109978

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child296 child278 - child
    bread379 bread341 - bread-portion
    content221 content59 - content-portion
    tray149 tray251 - tray
    table150 table487 table465 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray149 kitchen)
     (at tray251 kitchen)
     (at_kitchen_bread bread379)
     (at_kitchen_bread bread341)
     (at_kitchen_content content221)
     (at_kitchen_content content59)
     (not_allergic_gluten child296)
     (not_allergic_gluten child278)
     (waiting child296 table487)
     (waiting child278 table487)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child296)
     (served child278)
    )
  )
)
