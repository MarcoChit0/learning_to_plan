; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 110202

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child318 - child
    bread364 - bread-portion
    content250 - content-portion
    tray444 tray313 - tray
    table465 table54 table235 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray444 kitchen)
     (at tray313 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_content content250)
     (not_allergic_gluten child318)
     (waiting child318 table465)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child318)
    )
  )
)
