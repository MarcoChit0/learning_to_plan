; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 10491

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child51 child463 - child
    bread355 bread20 - bread-portion
    content389 content16 - content-portion
    tray386 tray192 tray466 - tray
    table178 table131 table74 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray386 kitchen)
     (at tray192 kitchen)
     (at tray466 kitchen)
     (at_kitchen_bread bread355)
     (at_kitchen_bread bread20)
     (at_kitchen_content content389)
     (at_kitchen_content content16)
     (not_allergic_gluten child51)
     (not_allergic_gluten child463)
     (waiting child51 table178)
     (waiting child463 table131)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child51)
     (served child463)
    )
  )
)
