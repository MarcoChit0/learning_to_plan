; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 92942

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child321 - child
    bread322 - bread-portion
    content104 - content-portion
    tray200 tray138 tray108 tray173 - tray
    table84 table459 table313 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at tray138 kitchen)
     (at tray108 kitchen)
     (at tray173 kitchen)
     (at_kitchen_bread bread322)
     (at_kitchen_content content104)
     (not_allergic_gluten child321)
     (waiting child321 table313)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child321)
    )
  )
)
