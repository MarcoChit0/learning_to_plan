; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 612207

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child443 - child
    bread178 - bread-portion
    content435 - content-portion
    tray47 tray147 tray85 - tray
    table65 table244 table390 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray47 kitchen)
     (at tray147 kitchen)
     (at tray85 kitchen)
     (at_kitchen_bread bread178)
     (at_kitchen_content content435)
     (not_allergic_gluten child443)
     (waiting child443 table244)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child443)
    )
  )
)
