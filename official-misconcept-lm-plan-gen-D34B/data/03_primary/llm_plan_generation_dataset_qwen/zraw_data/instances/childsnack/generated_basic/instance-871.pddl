; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 445516

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child371 child15 - child
    bread277 bread0 - bread-portion
    content209 content126 - content-portion
    tray167 tray218 - tray
    table253 table425 table368 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at tray218 kitchen)
     (at_kitchen_bread bread277)
     (at_kitchen_bread bread0)
     (at_kitchen_content content209)
     (at_kitchen_content content126)
     (no_gluten_bread bread277)
     (no_gluten_content content126)
     (allergic_gluten child15)
     (not_allergic_gluten child371)
     (waiting child371 table425)
     (waiting child15 table253)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child371)
     (served child15)
    )
  )
)
