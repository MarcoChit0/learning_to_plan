; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 166117

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child143 child358 - child
    bread137 bread273 - bread-portion
    content232 content321 - content-portion
    tray144 tray383 tray307 - tray
    table95 table449 table288 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray144 kitchen)
     (at tray383 kitchen)
     (at tray307 kitchen)
     (at_kitchen_bread bread137)
     (at_kitchen_bread bread273)
     (at_kitchen_content content232)
     (at_kitchen_content content321)
     (not_allergic_gluten child143)
     (not_allergic_gluten child358)
     (waiting child143 table95)
     (waiting child358 table288)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child143)
     (served child358)
    )
  )
)
