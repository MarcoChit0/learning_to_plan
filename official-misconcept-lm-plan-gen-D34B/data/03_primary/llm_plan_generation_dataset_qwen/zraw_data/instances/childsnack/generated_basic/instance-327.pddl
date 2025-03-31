; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 990826

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child463 - child
    bread200 - bread-portion
    content53 - content-portion
    tray352 tray325 tray341 tray120 - tray
    table197 table343 table65 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray352 kitchen)
     (at tray325 kitchen)
     (at tray341 kitchen)
     (at tray120 kitchen)
     (at_kitchen_bread bread200)
     (at_kitchen_content content53)
     (not_allergic_gluten child463)
     (waiting child463 table65)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child463)
    )
  )
)
