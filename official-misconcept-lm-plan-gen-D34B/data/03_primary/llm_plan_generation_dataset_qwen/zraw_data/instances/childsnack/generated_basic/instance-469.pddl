; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 128334

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child456 child298 child377 - child
    bread198 bread322 bread194 - bread-portion
    content140 content307 content353 - content-portion
    tray426 - tray
    table98 table191 table39 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray426 kitchen)
     (at_kitchen_bread bread198)
     (at_kitchen_bread bread322)
     (at_kitchen_bread bread194)
     (at_kitchen_content content140)
     (at_kitchen_content content307)
     (at_kitchen_content content353)
     (not_allergic_gluten child456)
     (not_allergic_gluten child298)
     (not_allergic_gluten child377)
     (waiting child456 table98)
     (waiting child298 table191)
     (waiting child377 table191)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child456)
     (served child298)
     (served child377)
    )
  )
)
